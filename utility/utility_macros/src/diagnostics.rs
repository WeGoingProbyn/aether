// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! `#[derive(StateDiagnostics)]` proc-macro.
//!
//! Reads a single helper attribute `#[diagnostics(...)]` and emits up to
//! three trait impls on the target type:
//!
//! - `Diagnostics<N>`             — from `components(...)` (required).
//! - `ConservationQuantities<N>`  — from `conserved(...)` (optional).
//! - `ExtraDiagnostics<N>`        — from `extras(...)` (optional).
//!
//! `N` is inferred from the count of `components(...)` entries. The
//! parser is a small hand-rolled recursive-descent over `(name, payload)`
//! tuples; `payload` is a string for `components`, an integer for
//! `conserved`, and an arbitrary expression for `extras`.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{DeriveInput, Expr, Ident, LitInt, LitStr, Token};

struct DiagnosticsAttr {
  components: Vec<LitStr>,
  conserved: Option<Vec<(LitStr, LitInt)>>,
  extras: Option<Vec<(LitStr, Expr)>>,
}

struct Tuple2<A, B>(A, B);

impl<A: Parse, B: Parse> Parse for Tuple2<A, B> {
  fn parse(input: ParseStream) -> syn::Result<Self> {
    let inner;
    syn::parenthesized!(inner in input);
    let a: A = inner.parse()?;
    let _: Token![,] = inner.parse()?;
    let b: B = inner.parse()?;
    Ok(Tuple2(a, b))
  }
}

impl Parse for DiagnosticsAttr {
  fn parse(input: ParseStream) -> syn::Result<Self> {
    let mut components: Option<Vec<LitStr>> = None;
    let mut conserved: Option<Vec<(LitStr, LitInt)>> = None;
    let mut extras: Option<Vec<(LitStr, Expr)>> = None;

    while !input.is_empty() {
      let key: Ident = input.parse()?;
      let inner;
      syn::parenthesized!(inner in input);

      match key.to_string().as_str() {
        "components" => {
          let names: Punctuated<LitStr, Token![,]> =
            Punctuated::parse_terminated(&inner)?;
          components = Some(names.into_iter().collect());
        }
        "conserved" => {
          let entries: Punctuated<Tuple2<LitStr, LitInt>, Token![,]> =
            Punctuated::parse_terminated(&inner)?;
          conserved = Some(entries.into_iter().map(|t| (t.0, t.1)).collect());
        }
        "extras" => {
          let entries: Punctuated<Tuple2<LitStr, Expr>, Token![,]> =
            Punctuated::parse_terminated(&inner)?;
          extras = Some(entries.into_iter().map(|t| (t.0, t.1)).collect());
        }
        other => {
          return Err(syn::Error::new(
            key.span(),
            format!(
              "unknown diagnostics clause `{}`; \
               expected `components`, `conserved`, or `extras`",
              other
            ),
          ));
        }
      }

      if input.is_empty() {
        break;
      }
      input.parse::<Token![,]>()?;
    }

    let components = components.ok_or_else(|| {
      syn::Error::new(
        Span::call_site(),
        "#[diagnostics(...)] requires a `components(...)` clause",
      )
    })?;

    Ok(DiagnosticsAttr {
      components,
      conserved,
      extras,
    })
  }
}

pub fn expand(input: TokenStream) -> TokenStream {
  let input = syn::parse::<DeriveInput>(input).unwrap();
  let name = &input.ident;
  let (impl_generics, ty_generics, where_clause) =
    input.generics.split_for_impl();

  let attr = match input
    .attrs
    .iter()
    .find(|a| a.path().is_ident("diagnostics"))
  {
    Some(a) => a,
    None => {
      return syn::Error::new(
        Span::call_site(),
        "#[derive(StateDiagnostics)] requires a `#[diagnostics(...)]` \
         attribute on the same item",
      )
      .to_compile_error()
      .into();
    }
  };

  let parsed: DiagnosticsAttr = match attr.parse_args() {
    Ok(p) => p,
    Err(e) => return e.to_compile_error().into(),
  };

  let n = parsed.components.len();
  let component_names = &parsed.components;

  let mut output = quote! {
    impl #impl_generics ::utility::diagnostics::Diagnostics<#n>
      for #name #ty_generics #where_clause
    {
      const COMPONENT_NAMES: [&'static str; #n] = [#(#component_names),*];
    }
  };

  if let Some(conserved) = parsed.conserved {
    let entries = conserved.iter().map(|(name, comp)| {
      quote! {
        ::utility::diagnostics::ConservedQuantity {
          name: #name,
          component: #comp,
        }
      }
    });
    output.extend(quote! {
      impl #impl_generics ::utility::diagnostics::ConservationQuantities<#n>
        for #name #ty_generics #where_clause
      {
        const CONSERVED_QUANTITIES:
          &'static [::utility::diagnostics::ConservedQuantity<#n>] = &[
          #(#entries),*
        ];
      }
    });
  }

  if let Some(extras) = parsed.extras {
    let extras_count = extras.len();
    let entries = extras.iter().map(|(name, expr)| {
      quote! { (#name, #expr) }
    });
    output.extend(quote! {
      impl #impl_generics ::utility::diagnostics::ExtraDiagnostics<#n>
        for #name #ty_generics #where_clause
      {
        fn extras<'__diag_a>(
          &'__diag_a self,
          state: &'__diag_a [f64; #n],
        ) -> impl ::core::iter::Iterator<Item = (&'static str, f64)>
          + '__diag_a
        {
          let entries: [(&'static str, f64); #extras_count] = [
            #(#entries),*
          ];
          entries.into_iter()
        }
      }
    });
  }

  output.into()
}
