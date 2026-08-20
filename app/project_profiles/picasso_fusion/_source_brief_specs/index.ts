import { BriefSpec } from './types';
import realEstateListingSheet from './real-estate-listing-sheet.json';
import realEstateFlyerBrochure from './real-estate-flyer-brochure.json';
import realEstateMarketReport from './real-estate-market-report.json';
import realEstateBuyerSellerGuide from './real-estate-buyer-seller-guide.json';
import realEstateCmaPresentation from './real-estate-cma-presentation.json';
import realEstateTestimonialSet from './real-estate-testimonial-set.json';
import realEstateEmailTemplate from './real-estate-email-template.json';
import realEstateLinkedinBanner from './real-estate-linkedin-banner.json';
import realEstateYoutubeBanner from './real-estate-youtube-banner.json';
import realEstateOpenHomeSignboard from './real-estate-open-home-signboard.json';
import startupEntrepreneurPitchDeckInvestor from './startup-entrepreneur-startup-pitch-deck-investor.json';
import startupEntrepreneurSalesDeck from './startup-entrepreneur-sales-deck.json';
import startupEntrepreneurPresentationTemplate from './startup-entrepreneur-presentation-template.json';
import startupEntrepreneurOnePager from './startup-entrepreneur-one-pager.json';
import startupEntrepreneurProposalTemplate from './startup-entrepreneur-proposal-template.json';
import startupEntrepreneurPressKit from './startup-entrepreneur-press-kit.json';
import startupEntrepreneurEmailSignature from './startup-entrepreneur-email-signature.json';
import startupEntrepreneurLinkedinBanner from './startup-entrepreneur-linkedin-banner.json';
import startupEntrepreneurBusinessCard from './startup-entrepreneur-business-card.json';
import technologyServicesCaseStudy from './technology-services-case-study.json';
import technologyServicesWhitePaper from './technology-services-white-paper.json';
import technologyServicesProductOnePager from './technology-services-product-one-pager.json';
import technologyServicesLinkedinBanner from './technology-services-linkedin-banner.json';
import technologyServicesEmailTemplate from './technology-services-email-template.json';
import technologyServicesPitchDeckInvestor from './technology-services-pitch-deck-investor.json';
import technologyServicesSalesDeck from './technology-services-sales-deck.json';
import technologyServicesWebBannerSet from './technology-services-web-banner-set.json';
import fitnessWellnessStaticPostSet from './fitness-wellness-static-post-set.json';
import fitnessWellnessCarouselSet from './fitness-wellness-carousel-set.json';
import fitnessWellnessReel from './fitness-wellness-reel.json';
import fitnessWellnessPrintAddon from './fitness-wellness-print-addon.json';
import fitnessWellnessVideoAddon from './fitness-wellness-video-addon.json';
import fitnessWellnessDigitalAddon from './fitness-wellness-digital-addon.json';
import ecommerceStaticPost from './ecommerce-static-post.json';
import ecommerceCarousel from './ecommerce-carousel.json';
import ecommerceReel from './ecommerce-reel.json';
import ecommercePrintAddon from './ecommerce-print-addon.json';
import ecommerceVideoAddon from './ecommerce-video-addon.json';
import ecommerceDigitalAddon from './ecommerce-digital-addon.json';
import ecommerceProductMockup from './ecommerce-product-mockup.json';
import ecommerceAdBanner from './ecommerce-ad-banner.json';
import retailBoutiqueStaticPost from './retail-boutique-static-post.json';
import retailBoutiqueCarousel from './retail-boutique-carousel.json';
import retailBoutiqueReel from './retail-boutique-reel.json';
import retailBoutiquePrintAddon from './retail-boutique-print-addon.json';
import retailBoutiqueVideoAddon from './retail-boutique-video-addon.json';
import retailBoutiqueDigitalAddon from './retail-boutique-digital-addon.json';
import restaurantCafeStaticPost from './restaurant-cafe-static-post.json';
import restaurantCafeCarousel from './restaurant-cafe-carousel.json';
import restaurantCafeReel from './restaurant-cafe-reel.json';
import restaurantCafePrintAddon from './restaurant-cafe-print-addon.json';
import restaurantCafeVideoAddon from './restaurant-cafe-video-addon.json';
import restaurantCafeDigitalAddon from './restaurant-cafe-digital-addon.json';

// Register additional brief-type specs here — each is a standalone JSON file
// under this directory (see types.ts for the shape). No DB migration or
// admin-UI wiring needed: ai-chat.service.ts picks these up purely by
// matching vertical/template name and folds them into its system prompt.
const SPECS: BriefSpec[] = [
  realEstateListingSheet as BriefSpec,
  realEstateFlyerBrochure as BriefSpec,
  realEstateMarketReport as BriefSpec,
  realEstateBuyerSellerGuide as BriefSpec,
  realEstateCmaPresentation as BriefSpec,
  realEstateTestimonialSet as BriefSpec,
  realEstateEmailTemplate as BriefSpec,
  realEstateLinkedinBanner as BriefSpec,
  realEstateYoutubeBanner as BriefSpec,
  realEstateOpenHomeSignboard as BriefSpec,
  startupEntrepreneurPitchDeckInvestor as BriefSpec,
  startupEntrepreneurSalesDeck as BriefSpec,
  startupEntrepreneurPresentationTemplate as BriefSpec,
  startupEntrepreneurOnePager as BriefSpec,
  startupEntrepreneurProposalTemplate as BriefSpec,
  startupEntrepreneurPressKit as BriefSpec,
  startupEntrepreneurEmailSignature as BriefSpec,
  startupEntrepreneurLinkedinBanner as BriefSpec,
  startupEntrepreneurBusinessCard as BriefSpec,
  technologyServicesCaseStudy as BriefSpec,
  technologyServicesWhitePaper as BriefSpec,
  technologyServicesProductOnePager as BriefSpec,
  technologyServicesLinkedinBanner as BriefSpec,
  technologyServicesEmailTemplate as BriefSpec,
  technologyServicesPitchDeckInvestor as BriefSpec,
  technologyServicesSalesDeck as BriefSpec,
  technologyServicesWebBannerSet as BriefSpec,
  fitnessWellnessStaticPostSet as BriefSpec,
  fitnessWellnessCarouselSet as BriefSpec,
  fitnessWellnessReel as BriefSpec,
  fitnessWellnessPrintAddon as BriefSpec,
  fitnessWellnessVideoAddon as BriefSpec,
  fitnessWellnessDigitalAddon as BriefSpec,
  ecommerceStaticPost as BriefSpec,
  ecommerceCarousel as BriefSpec,
  ecommerceReel as BriefSpec,
  ecommercePrintAddon as BriefSpec,
  ecommerceVideoAddon as BriefSpec,
  ecommerceDigitalAddon as BriefSpec,
  ecommerceProductMockup as BriefSpec,
  ecommerceAdBanner as BriefSpec,
  retailBoutiqueStaticPost as BriefSpec,
  retailBoutiqueCarousel as BriefSpec,
  retailBoutiqueReel as BriefSpec,
  retailBoutiquePrintAddon as BriefSpec,
  retailBoutiqueVideoAddon as BriefSpec,
  retailBoutiqueDigitalAddon as BriefSpec,
  restaurantCafeStaticPost as BriefSpec,
  restaurantCafeCarousel as BriefSpec,
  restaurantCafeReel as BriefSpec,
  restaurantCafePrintAddon as BriefSpec,
  restaurantCafeVideoAddon as BriefSpec,
  restaurantCafeDigitalAddon as BriefSpec,
];

export function findBriefSpec(
  verticalKey: string | undefined,
  // Any strings that might name the specific brief type: the BriefTemplate
  // name AND, for addon-scoped briefs, the addon order's product description
  // (e.g. "Property flyer / brochure (2pp)") — for addon orders,
  // template.name is only the generic vertical-wide template, so the
  // specific product name lives on the order, not the template. Pass both;
  // whichever one actually contains a matching keyword wins.
  ...candidateNames: Array<string | null | undefined>
): BriefSpec | undefined {
  const nameLower = candidateNames.filter(Boolean).join(' ').toLowerCase();

  // Template/product name is the specific signal — check it first so two
  // specs that share the same vertical (e.g. a listing sheet vs. a
  // flyer/brochure, both "realestate") never get confused with one another.
  // Search within the given vertical first: different verticals can
  // legitimately reuse the same templateNameMatch keywords (e.g. "linkedin
  // banner" exists for both "realestate" and "startup"), and a bare
  // cross-vertical search would always resolve to array order.
  const inVertical = verticalKey
    ? SPECS.filter((spec) => spec.verticalKey === verticalKey)
    : SPECS;
  const byNameInVertical = inVertical.find((spec) =>
    spec.templateNameMatch?.some((m) => nameLower.includes(m)),
  );
  if (byNameInVertical) return byNameInVertical;

  // Cross-vertical name fallback — only safe when exactly one spec anywhere
  // claims this name, otherwise a shared keyword risks silently attaching
  // the wrong vertical's checklist.
  const byNameAnywhere = SPECS.filter((spec) =>
    spec.templateNameMatch?.some((m) => nameLower.includes(m)),
  );
  if (byNameAnywhere.length === 1) return byNameAnywhere[0];

  // Bare vertical match is only safe as a fallback when exactly one spec
  // claims that vertical — with multiple specs per vertical, an ambiguous
  // vertical-only match risks silently attaching the wrong checklist.
  if (!verticalKey) return undefined;
  const verticalMatches = SPECS.filter(
    (spec) => spec.verticalKey === verticalKey,
  );
  return verticalMatches.length === 1 ? verticalMatches[0] : undefined;
}

// Renders a spec as a compact checklist for the system prompt — one line per
// question, code first so the model can copy it verbatim as the "code" field
// on its reply (see BRIEF_REPLY_TOOL in ai-chat.service.ts).
export function formatBriefSpecChecklist(spec: BriefSpec): string {
  return spec.sections
    .map((section) => {
      const lines = section.questions
        .map((q) => {
          const required = q.required === false ? '[OPTIONAL]' : '[REQUIRED]';
          const opts = q.options?.length
            ? ` | Options: ${q.options.map((o) => o.label).join(', ')}`
            : '';
          const hint = q.hint ? ` — ${q.hint}` : '';
          return `  - [${q.code}] (${q.kind}) ${q.question}${hint}${opts} ${required}`;
        })
        .join('\n');
      return `### ${section.name}\n${lines}`;
    })
    .join('\n\n');
}
